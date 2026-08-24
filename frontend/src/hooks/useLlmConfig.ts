import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  ConnectionUpsertPayload,
  LlmConfigState,
  RoleAssignPayload,
  LlmRole,
} from "@/types/llmConfig";

/** GET /admin/llm-config — role assignments + connections (V2). */
export function useLlmConfig() {
  return useQuery({
    queryKey: ["llm-config"],
    queryFn: () => api.get<LlmConfigState>("/admin/llm-config"),
  });
}

/** PUT /admin/llm-config/{role} — assign {conn_id, model}. */
export function useUpdateLlmConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ role, data }: { role: LlmRole; data: RoleAssignPayload }) =>
      api.updateLlmConfig(role, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-config"] });
    },
  });
}

/** DELETE /admin/llm-config/{role} — clear assignment → back to .env. */
export function useDeleteLlmConfigOverride() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (role: LlmRole) => api.deleteLlmConfigOverride(role),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-config"] });
    },
  });
}

/** PUT /admin/llm-config/connections/{conn_id} — create or update. */
export function useSaveConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      connId,
      data,
    }: {
      connId: string;
      data: ConnectionUpsertPayload;
    }) => api.saveConnection(connId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-config"] });
    },
  });
}

/** DELETE /admin/llm-config/connections/{conn_id}?force= */
export function useDeleteConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ connId, force }: { connId: string; force?: boolean }) =>
      api.deleteConnection(connId, force),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-config"] });
    },
  });
}
