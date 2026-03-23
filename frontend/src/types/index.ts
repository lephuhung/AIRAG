// ── Auth Types ──
export interface User {
  id: number;
  email: string;
  full_name: string;
  is_active: boolean;
  is_superadmin: boolean;
  avatar_url?: string | null;
  created_at: string;
  updated_at: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

export interface Tenant {
  id: number;
  name: string;
  slug: string;
  domain: string | null;
  is_active: boolean;
  member_count: number;
  pending_count: number;
  created_at: string;
  updated_at: string;
}

export interface TenantUser {
  id: number;
  tenant_id: number;
  user_id: number;
  role: string;
  is_approved: boolean;
  created_at: string;
  email: string | null;
  full_name: string | null;
  tenant_name: string | null;
}

// ── Admin Types ──
export interface AdminUserDetail extends User {
  tenant_memberships: TenantUser[];
}

export interface AdminUserListResponse {
  users: AdminUserDetail[];
  total: number;
  page: number;
  per_page: number;
}

export interface DocumentTypeBreakdown {
  name: string;
  count: number;
}

export interface DateCount {
  date: string;
  count: number;
}

export interface DocumentStatusBreakdown {
  status: string;
  count: number;
}

export interface TopWorkspace {
  id: number;
  name: string;
  total_size: number;
  doc_count: number;
}

export interface FailedDocument {
  id: number;
  filename: string;
  workspace_name: string;
  error_message: string | null;
}

export interface PendingApproval {
  user_id: number;
  email: string;
  tenant_name: string;
  role: string;
}

export interface AdminStats {
  total_users: number;
  active_users: number;
  pending_users: number;
  total_tenants: number;
  total_documents: number;
  total_knowledge_bases: number;
  document_type_breakdown: DocumentTypeBreakdown[];
  users_growth: DateCount[];
  chat_growth: DateCount[];
  document_status_breakdown: DocumentStatusBreakdown[];
  top_workspaces: TopWorkspace[];
  recent_failed_docs: FailedDocument[];
  pending_approvals: PendingApproval[];
}

// Knowledge Base (Document Workspace)
export interface KnowledgeBase {
  id: number;
  name: string;
  description: string | null;
  system_prompt: string | null;
  document_count: number;
  indexed_count: number;
  created_at: string;
  updated_at: string;
  visibility: "public" | "tenant" | "personal";
  owner_id: number | null;
  tenant_id: number | null;
}

export interface CreateWorkspace {
  name: string;
  description?: string;
  visibility?: "public" | "tenant" | "personal";
  tenant_id?: number | null;
}

export interface UpdateWorkspace {
  name?: string;
  description?: string;
  system_prompt?: string | null;
  tenant_id?: number | null;
  visibility?: "public" | "tenant" | "personal";
}

export interface WorkspaceSummary {
  id: number;
  name: string;
  document_count: number;
}

export interface DocumentTypeInfo {
  id: number;
  slug: string;
  name: string;
}

export interface Document {
  id: number;
  workspace_id: number;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  status: DocumentStatus;
  chunk_count: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  // HRAG metadata
  page_count?: number;
  image_count?: number;
  table_count?: number;
  parser_version?: string;           // "docling" | "legacy"
  processing_time_ms?: number;
  // Digital signature metadata (PDF only, null if no signatures found)
  digital_signatures?: Array<{
    field_name?: string;
    page?: number;
    signer_name?: string;
    organization?: string;
    email?: string;
    issuer?: string;
    valid_from?: string;
    valid_until?: string;
    signing_time?: string;
    reason?: string;
    location?: string;
  }> | null;
  // Document type classification (auto-detected by pipeline)
  document_type_id?: number | null;
  document_type?: DocumentTypeInfo | null;
  // Sub-task completion flags (set independently by each worker after CHUNKING)
  embed_done?: boolean;              // embed worker → ChromaDB done
  captions_done?: boolean;           // caption worker → image/table captions done
  kg_done?: boolean;                 // kg worker → LightRAG KG ingest done
}

// RAG Types
export type DocumentStatus =
  | "pending"          // uploaded, waiting for parse_worker
  | "parsing"          // parse_worker: Docling on native docs
  | "ocring"           // parse_worker: OCR on scanned PDFs
  | "chunking"         // parse done → embed+caption+kg dispatched
  | "embedding"        // embed_worker running
  | "building_kg"      // embed+captions done, KG still running
  | "indexed"          // all done
  | "failed";

export type RAGQueryMode = "hybrid" | "vector_only" | "naive" | "local" | "global";

export interface RAGQueryRequest {
  question: string;
  top_k?: number;
  document_ids?: number[];
  mode?: RAGQueryMode;
}

export interface Citation {
  source_file: string;
  document_id: number | null;
  page_no: number | null;
  heading_path: string[];
  formatted: string;
}

export interface DocumentImage {
  image_id: string;
  document_id: number;
  page_no: number;
  caption: string;
  width: number;
  height: number;
  url: string;
}

export interface RetrievedChunk {
  content: string;
  chunk_id: string;
  score: number;
  metadata: Record<string, unknown>;
  citation?: Citation;
}

export interface RAGQueryResponse {
  query: string;
  chunks: RetrievedChunk[];
  context: string;
  total_chunks: number;
  knowledge_graph_summary?: string;
  citations?: Citation[];
  image_refs?: DocumentImage[];
}

export interface RAGStats {
  workspace_id: number;
  total_documents: number;
  indexed_documents: number;
  total_chunks: number;
  image_count?: number;
  hrag_documents?: number;
}

// Knowledge Graph Types
export interface KGEntity {
  name: string;
  entity_type: string;
  description: string;
  degree: number;
}

export interface KGRelationship {
  source: string;
  target: string;
  description: string;
  keywords: string;
  weight: number;
}

export interface KGGraphNode {
  id: string;
  label: string;
  entity_type: string;
  degree: number;
}

export interface KGGraphEdge {
  source: string;
  target: string;
  label: string;
  weight: number;
}

export interface KGGraphData {
  nodes: KGGraphNode[];
  edges: KGGraphEdge[];
  is_truncated: boolean;
}

export interface KGAnalytics {
  entity_count: number;
  relationship_count: number;
  entity_types: Record<string, number>;
  top_entities: KGEntity[];
  avg_degree: number;
}

export interface DocumentBreakdown {
  document_id: number;
  filename: string;
  chunk_count: number;
  image_count: number;
  page_count: number;
  file_size: number;
  status: string;
}

export interface ProjectAnalytics {
  stats: RAGStats;
  kg_analytics: KGAnalytics | null;
  document_breakdown: DocumentBreakdown[];
}

// Chat Types
export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatImageRef {
  ref_id?: string;  // 4-char alphanumeric ID, e.g. "p4f2"
  image_id: string;
  document_id: number;
  page_no: number;
  caption: string;
  url: string;
  width: number;
  height: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSourceChunk[];
  relatedEntities?: string[];
  imageRefs?: ChatImageRef[];
  thinking?: string | null;
  timestamp: string;
  isStreaming?: boolean;
  agentSteps?: AgentStep[];
}

export interface ChatSourceChunk {
  index: number | string;  // number for legacy, string for new [a3x9] format
  chunk_id: string;
  content: string;
  document_id: number;
  page_no: number;
  heading_path: string[];
  score: number;
  source_type?: "vector" | "kg";
}

export interface ChatResponseData {
  answer: string;
  sources: ChatSourceChunk[];
  related_entities: string[];
  kg_summary: string | null;
  image_refs: ChatImageRef[];
  thinking: string | null;
}

export interface PersistedChatMessage {
  id: number;
  message_id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSourceChunk[] | null;
  related_entities?: string[] | null;
  image_refs?: ChatImageRef[] | null;
  thinking?: string | null;
  agent_steps?: AgentStep[] | null;
  created_at: string;
}

export interface ChatHistoryResponse {
  session_id?: string;
  messages: PersistedChatMessage[];
  total: number;
}

export interface LLMCapabilities {
  provider: string;
  model: string;
  supports_thinking: boolean;
  supports_vision: boolean;
  thinking_default: boolean;
}

// SSE Streaming Types
// * useRAGChatStream — SSE streaming hook for HRAG chat.
export type ChatStreamStatus = "idle" | "analyzing" | "retrieving" | "generating" | "error";

// Agent Step Types (ThinkingTimeline)
export type AgentStepType =
  | "analyzing"
  | "understood"
  | "retrieving"
  | "sources_found"
  | "generating"
  | "done"
  | "error";

export type AgentStepStatus = "active" | "completed" | "error";

export interface AgentStep {
  id: string;
  step: AgentStepType;
  detail: string;
  status: AgentStepStatus;
  timestamp: number;
  durationMs?: number;
  thinkingText?: string;
  sourceBadges?: string[];
  sourceCount?: number;
  imageCount?: number;
}

// Worker Management Types
export interface QueueInfo {
  name: string;
  messages_ready: number;
  messages_unacked: number;
  consumers: number;
  message_rate_in: number;
  message_rate_out: number;
  has_dlx?: boolean;
}

export interface PipelineSummary {
  pending: number;
  parsing: number;
  ocring: number;
  chunking: number;
  embedding: number;
  building_kg: number;
  indexed: number;
  failed: number;
}

export interface WorkerOverview {
  queues: QueueInfo[];
  pipeline_summary: PipelineSummary;
  active_workers: Record<string, number>;
  managed_workers: Record<string, number>;
  rabbitmq_connected: boolean;
}

export interface PipelineDocument {
  id: number;
  filename: string;
  workspace_id: number;
  status: DocumentStatus;
  embed_done: boolean;
  captions_done: boolean;
  kg_done: boolean;
  processing_time_ms: number;
  error_message: string | null;
  updated_at: string;
}

// ── Worker Health Check ──
export interface WorkerHealthCheck {
  status: "healthy" | "degraded" | "unhealthy";
  checks: {
    rabbitmq: {
      status: string;
      version?: string;
      cluster?: string;
      error?: string;
      queue_totals?: Record<string, number>;
    };
    queues: Record<string, {
      status: string;
      consumers: number;
      messages_ready: number;
      messages_unacked: number;
      has_dlx: boolean;
      warnings: string[];
    }>;
    dead_letter_queue: {
      status: string;
      messages: number;
    };
    managed_workers: Record<string, {
      running: number;
      total_spawned: number;
      pids: number[];
    }>;
    pipeline: {
      status: string;
      documents_in_progress: number;
      documents_failed: number;
    };
  };
}

// ── Managed Worker Process ──
export interface ManagedWorkerInfo {
  worker_type: string;
  pid: number | null;
  alive: boolean;
  started_at: number;
  uptime_seconds: number;
  restart_count: number;
  return_code: number | null;
}

// ── Dead Letter Queue ──
export interface DeadLetterMessage {
  payload: string;
  headers: Record<string, unknown>;
  exchange: string;
  routing_key: string;
  redelivered: boolean;
}

// ── Document Type Admin Types ──
export interface DocumentTypeDetail {
  id: number;
  slug: string;
  name: string;
  description: string | null;
  is_active: boolean;
}

export interface DocumentTypeSystemPromptResponse {
  document_type_slug: string;
  workspace_id: number | null;
  system_prompt: string;
  is_default: boolean;
}

// ── Invite Link Types ──
export interface InviteValidation {
  valid: boolean;
  tenant_name: string | null;
  tenant_slug: string | null;
  email: string | null;
  expires_at: string | null;
}

export interface InviteLink {
  id: number;
  token: string;
  tenant_id: number;
  email: string | null;
  role: string;
  max_uses: number | null;
  use_count: number;
  expires_at: string;
  created_at: string;
  is_active: boolean;
  invite_url: string;
}
