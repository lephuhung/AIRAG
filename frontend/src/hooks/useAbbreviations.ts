import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { 
  Abbreviation, 
  AbbreviationCreate, 
  AbbreviationUpdate, 
  AbbreviationListResponse 
} from "@/types";

export function useAbbreviations(
  search?: string,
  isActive?: boolean | null,
  page: number = 1,
  perPage: number = 20
) {
  return useQuery({
    queryKey: ["abbreviations", search, isActive, page, perPage],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (search) params.append("search", search);
      if (isActive !== null && isActive !== undefined) params.append("is_active", String(isActive));
      params.append("page", String(page));
      params.append("per_page", String(perPage));

      const res = await api.get<AbbreviationListResponse>(`/abbreviations/?${params.toString()}`);
      return res;
    },
  });
}

export function useCreateAbbreviation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: AbbreviationCreate) => {
      const res = await api.post<Abbreviation>("/abbreviations/", data);
      return res;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["abbreviations"] });
    },
  });
}

export function useUpdateAbbreviation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, data }: { id: number; data: AbbreviationUpdate }) => {
      const res = await api.patch<Abbreviation>(`/abbreviations/${id}`, data);
      return res;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["abbreviations"] });
    },
  });
}

export function useDeleteAbbreviation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: number) => {
      await api.delete(`/abbreviations/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["abbreviations"] });
    },
  });
}
