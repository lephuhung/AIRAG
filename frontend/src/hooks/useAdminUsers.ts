import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AdminUserListResponse, AdminUserDetail, AdminStats } from "@/types";

export function useAdminUsers(
  search?: string,
  isActive?: boolean | null,
  tenantId?: string | null,
  page: number = 1,
  perPage: number = 20,
) {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  if (isActive !== null && isActive !== undefined) params.set("is_active", String(isActive));
  if (tenantId) params.set("tenant_id", String(tenantId));
  params.set("page", String(page));
  params.set("per_page", String(perPage));

  return useQuery({
    queryKey: ["admin-users", search, isActive, tenantId, page, perPage],
    queryFn: () => api.get<AdminUserListResponse>(`/admin/users?${params.toString()}`),
  });
}

export function useAdminUserDetail(userId: string) {
  return useQuery({
    queryKey: ["admin-user", userId],
    queryFn: () => api.get<AdminUserDetail>(`/admin/users/${userId}`),
    enabled: !!userId,
  });
}

export function useAdminStats() {
  return useQuery({
    queryKey: ["admin-stats"],
    queryFn: () => api.get<AdminStats>("/admin/stats"),
  });
}

export function useUpdateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      userId,
      data,
    }: {
      userId: string;
      data: { is_active?: boolean; is_superadmin?: boolean; full_name?: string };
    }) => api.put<AdminUserDetail>(`/admin/users/${userId}`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      queryClient.invalidateQueries({ queryKey: ["admin-stats"] });
    },
  });
}

export function useDeleteUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (userId: string) => api.delete(`/admin/users/${userId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      queryClient.invalidateQueries({ queryKey: ["admin-stats"] });
    },
  });
}

export function useResetUserPassword() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, newPassword }: { userId: string; newPassword: string }) =>
      api.post<AdminUserDetail>(`/admin/users/${userId}/reset-password`, {
        new_password: newPassword,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
    },
  });
}

export function useUpdateTenantMemberRole() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      tenantId,
      userId,
      role,
    }: {
      tenantId: string;
      userId: string;
      role: "admin" | "member";
    }) => api.put<any>(`/tenants/${tenantId}/users/${userId}/role`, { role }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
    },
  });
}
