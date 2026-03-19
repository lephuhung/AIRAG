import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Tenant } from "@/types";

export function useMyTenants() {
  return useQuery({
    queryKey: ["my-tenants"],
    queryFn: () => api.get<Tenant[]>("/tenants/my"),
  });
}
