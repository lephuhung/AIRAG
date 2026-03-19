import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Tenant } from "@/types";

export function useAdminTenants() {
  return useQuery({
    queryKey: ["admin-tenants"],
    queryFn: () => api.get<Tenant[]>("/tenants"),
  });
}

export function useCreateTenant() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; slug: string; domain?: string }) =>
      api.post<Tenant>("/tenants", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-tenants"] });
      queryClient.invalidateQueries({ queryKey: ["admin-stats"] });
    },
  });
}

export function useUpdateTenant() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      tenantId,
      data,
    }: {
      tenantId: number;
      data: { name?: string; slug?: string; domain?: string; is_active?: boolean };
    }) => api.put<Tenant>(`/tenants/${tenantId}`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-tenants"] });
    },
  });
}

export function useDeactivateTenant() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (tenantId: number) => api.delete(`/tenants/${tenantId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-tenants"] });
      queryClient.invalidateQueries({ queryKey: ["admin-stats"] });
    },
  });
}
