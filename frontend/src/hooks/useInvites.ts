import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { InviteLink, InviteValidation } from "@/types";

export function useCreateInvite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      tenantId,
      data,
    }: {
      tenantId: number;
      data: {
        email?: string | null;
        role?: string;
        max_uses?: number | null;
        expires_in_days?: number;
      };
    }) => api.post<InviteLink>(`/tenants/${tenantId}/invites`, data),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["tenant-invites", variables.tenantId],
      });
    },
  });
}

export function useTenantInvites(tenantId: number | null) {
  return useQuery({
    queryKey: ["tenant-invites", tenantId],
    queryFn: () => api.get<InviteLink[]>(`/tenants/${tenantId}/invites`),
    enabled: !!tenantId,
  });
}

export function useRevokeInvite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      tenantId,
      inviteId,
    }: {
      tenantId: number;
      inviteId: number;
    }) => api.delete(`/tenants/${tenantId}/invites/${inviteId}`),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["tenant-invites", variables.tenantId],
      });
    },
  });
}

export function useValidateInvite(token: string | null) {
  return useQuery({
    queryKey: ["invite-validation", token],
    queryFn: () => api.get<InviteValidation>(`/tenants/invite/${token}`),
    enabled: !!token,
    retry: false,
    staleTime: 60_000, // cache for 1 minute
  });
}
