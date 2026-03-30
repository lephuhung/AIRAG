import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { DocumentTypeDetail, DocumentTypeSystemPromptResponse } from "@/types";

// ── Queries ──────────────────────────────────────────────────────────────────

export function useDocumentTypes(includeInactive = false) {
  return useQuery({
    queryKey: ["document-types", includeInactive],
    queryFn: () =>
      api.get<DocumentTypeDetail[]>(
        `/document-types${includeInactive ? "?include_inactive=true" : ""}`,
      ),
  });
}

export function useDocumentType(slug: string) {
  return useQuery({
    queryKey: ["document-type", slug],
    queryFn: () => api.get<DocumentTypeDetail>(`/document-types/${slug}`),
    enabled: !!slug,
  });
}

export function useDocumentTypeGlobalPrompt(slug: string) {
  return useQuery({
    queryKey: ["document-type-prompt", slug],
    queryFn: () =>
      api.get<DocumentTypeSystemPromptResponse>(`/document-types/${slug}/prompt`),
    enabled: !!slug,
  });
}

// ── Mutations ─────────────────────────────────────────────────────────────────

export function useCreateDocumentType() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { slug: string; name: string; description?: string }) =>
      api.post<DocumentTypeDetail>("/document-types", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["document-types"] });
    },
  });
}

export function useUpdateDocumentType() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      slug,
      data,
    }: {
      slug: string;
      data: { name?: string; description?: string; is_active?: boolean };
    }) => api.put<DocumentTypeDetail>(`/document-types/${slug}`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["document-types"] });
    },
  });
}

export function useDeactivateDocumentType() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (slug: string) => api.delete(`/document-types/${slug}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["document-types"] });
    },
  });
}

export function useSetGlobalPrompt() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      slug,
      system_prompt,
      kg_system_prompt,
    }: {
      slug: string;
      system_prompt: string;
      kg_system_prompt?: string | null;
    }) =>
      api.put<DocumentTypeSystemPromptResponse>(
        `/document-types/${slug}/prompt`,
        { system_prompt, kg_system_prompt },
      ),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["document-type-prompt", variables.slug],
      });
    },
  });
}
