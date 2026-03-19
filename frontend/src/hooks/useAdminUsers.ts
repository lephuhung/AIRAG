import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AdminUserListResponse, AdminUserDetail, AdminStats } from "@/types";

export function useAdminUsers(
  search?: string,
  isActive?: boolean | null,
  page: number = 1,
  perPage: number = 20,
) {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  if (isActive !== null && isActive !== undefined) params.set("is_active", String(isActive));
  params.set("page", String(page));
  params.set("per_page", String(perPage));

  return useQuery({
    queryKey: ["admin-users", search, isActive, page, perPage],
    queryFn: () => api.get<AdminUserListResponse>(`/admin/users?${params.toString()}`),
  });
}

export function useAdminUserDetail(userId: number) {
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
      userId: number;
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
    mutationFn: (userId: number) => api.delete(`/admin/users/${userId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      queryClient.invalidateQueries({ queryKey: ["admin-stats"] });
    },
  });
}

export function useResetUserPassword() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, newPassword }: { userId: number; newPassword: string }) =>
      api.post<AdminUserDetail>(`/admin/users/${userId}/reset-password`, {
        new_password: newPassword,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
    },
  });
}
