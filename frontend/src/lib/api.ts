import { useAuthStore } from "@/stores/authStore";
import type {
  ConnectionDeleteResult,
  ConnectionSaveResult,
  ConnectionUpsertPayload,
  LlmConfigState,
  LlmModelsRequest,
  LlmModelsResponse,
  LlmRole,
  LlmTestRequest,
  LlmTestResult,
  RoleAssignPayload,
  RoleAssignResult,
} from "@/types/llmConfig";

const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

/**
 * Rewrite a presigned MinIO URL so it goes through the Vite dev proxy
 * instead of hitting localhost:9000 directly (which is blocked by CORS/network
 * in dev). In production (Docker/nginx) the URL is already reachable.
 *
 * localhost:9000/bucket/key?sig=... → /minio-direct/bucket/key?sig=...
 */
export function rewritePresignedUrl(url: string): string {
  try {
    const parsed = new URL(url);
    // Only rewrite when running via Vite dev server (same origin, no explicit host)
    if (parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1") {
      return `/minio-direct${parsed.pathname}${parsed.search}`;
    }
  } catch {
    // Not a full URL — leave as-is
  }
  return url;
}

function getAuthHeaders(): Record<string, string> {
  const token = useAuthStore.getState().token;
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
}

/**
 * Handle 401 responses: try to refresh token, retry request once.
 * If refresh fails, logout and redirect to login.
 */
async function handleUnauthorized(
  originalUrl: string,
  originalOptions?: RequestInit,
): Promise<Response | null> {
  const store = useAuthStore.getState();
  const refreshed = await store.refreshAccessToken();

  if (!refreshed) {
    // Redirect to login forcefully
    if (window.location.pathname !== "/login") {
      window.location.assign("/login");
      // Delay to allow browser navigation to take over before React Query catches an error
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    return null;
  }

  // Retry the original request with new token
  const retryHeaders = {
    ...originalOptions?.headers,
    ...getAuthHeaders(),
  };
  return fetch(originalUrl, {
    ...originalOptions,
    headers: retryHeaders,
  });
}

class ApiClient {
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const url = `${BASE_URL}${path}`;
    const mergedOptions: RequestInit = {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
        ...options?.headers,
      },
    };

    let response = await fetch(url, mergedOptions);

    // Handle 401 — try token refresh
    if (response.status === 401) {
      const retried = await handleUnauthorized(url, mergedOptions);
      if (retried) {
        response = retried;
      } else {
        throw new Error("Session expired. Please login again.");
      }
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  }

  get<T>(path: string) {
    return this.request<T>(path, { method: "GET" });
  }

  /** Fetch a plain-text (or markdown) response as a string. */
  async getText(path: string): Promise<string> {
    const url = `${BASE_URL}${path}`;
    let response = await fetch(url, {
      method: "GET",
      headers: getAuthHeaders(),
    });

    if (response.status === 401) {
      const retried = await handleUnauthorized(url, {
        method: "GET",
        headers: getAuthHeaders(),
      });
      if (retried) response = retried;
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }
    return response.text();
  }

  post<T>(path: string, data?: unknown) {
    return this.request<T>(path, {
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  put<T>(path: string, data?: unknown) {
    return this.request<T>(path, {
      method: "PUT",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  patch<T>(path: string, data: unknown) {
    return this.request<T>(path, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  delete<T = unknown>(path: string) {
    return this.request(path, { method: "DELETE" }) as Promise<T>;
  }

  async downloadFile(path: string, filename: string): Promise<void> {
    const response = await fetch(`${BASE_URL}${path}`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Download failed" }));
      throw new Error(error.detail || `Download Error: ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /** POST text to the TTS endpoint and return the audio as a Blob for playback. */
  async synthesizeSpeech(body: {
    text: string;
    voice?: string;
    speed?: number;
    pitch?: number;
  }): Promise<Blob> {
    const url = `${BASE_URL}/tts/synthesize`;
    const options: RequestInit = {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeaders() },
      body: JSON.stringify(body),
    };

    let response = await fetch(url, options);
    if (response.status === 401) {
      const retried = await handleUnauthorized(url, options);
      if (retried) response = retried;
    }
    if (!response.ok) {
      throw new Error(
        response.status === 503
          ? "Dịch vụ đọc văn bản hiện không khả dụng."
          : `TTS Error: ${response.status}`,
      );
    }
    return response.blob();
  }

  /** POST a recorded audio blob to the STT endpoint and return the transcript. */
  async transcribeAudio(
    blob: Blob,
    filename = "recording.webm",
  ): Promise<{ text: string; language?: string; duration?: number }> {
    const url = `${BASE_URL}/stt/transcribe`;
    const formData = new FormData();
    formData.append("file", blob, filename);
    // NOTE: do not set Content-Type — the browser adds the multipart boundary.
    const options: RequestInit = {
      method: "POST",
      headers: { ...getAuthHeaders() },
      body: formData,
    };

    let response = await fetch(url, options);
    if (response.status === 401) {
      const retried = await handleUnauthorized(url, options);
      if (retried) response = retried;
    }
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: `STT Error: ${response.status}` }));
      throw new Error(error.detail || `STT Error: ${response.status}`);
    }
    return response.json();
  }

  /**
   * POST a PDF to the OCR endpoint and return the extracted text.
   *
   * Uses XHR (not fetch) so the caller can show real upload progress and a
   * distinct "processing" phase once the bytes are uploaded and the server is
   * running OCR. Refreshes the token and retries once on 401.
   */
  async extractPdfText(
    file: File,
    opts?: {
      onUploadProgress?: (percent: number) => void;
      onProcessing?: () => void;
    },
  ): Promise<{ text: string; pages?: number }> {
    const url = `${BASE_URL}/ocr/extract`;

    const send = (): Promise<{ status: number; body: string }> =>
      new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append("file", file, file.name || "document.pdf");
        const xhr = new XMLHttpRequest();
        xhr.open("POST", url);
        // NOTE: do not set Content-Type — the browser adds the multipart boundary.
        for (const [k, v] of Object.entries(getAuthHeaders())) xhr.setRequestHeader(k, v);
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) opts?.onUploadProgress?.(Math.round((e.loaded / e.total) * 100));
        };
        // Bytes are uploaded — the server is now running OCR.
        xhr.upload.onload = () => opts?.onProcessing?.();
        xhr.onload = () => resolve({ status: xhr.status, body: xhr.responseText });
        xhr.onerror = () => reject(new Error("Network error during OCR upload"));
        xhr.send(formData);
      });

    let res = await send();
    if (res.status === 401) {
      const refreshed = await useAuthStore.getState().refreshAccessToken();
      if (refreshed) {
        res = await send();
      } else {
        if (window.location.pathname !== "/login") window.location.assign("/login");
        throw new Error("Unauthorized");
      }
    }
    if (res.status < 200 || res.status >= 300) {
      let detail = `OCR Error: ${res.status}`;
      try {
        detail = JSON.parse(res.body).detail || detail;
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }
    return JSON.parse(res.body);
  }

  // ── Admin: LLM runtime config V2 (/admin/llm-config) ───────────────────

  getLlmConfig() {
    return this.get<LlmConfigState>("/admin/llm-config");
  }

  /** Save a connection (endpoint + key). Blank api_key keeps the stored one. */
  saveConnection(connId: string, payload: ConnectionUpsertPayload) {
    return this.put<ConnectionSaveResult>(
      `/admin/llm-config/connections/${encodeURIComponent(connId)}`,
      payload,
    );
  }

  /** Delete a connection. force=true also clears referencing role assignments. */
  deleteConnection(connId: string, force = false) {
    return this.delete<ConnectionDeleteResult>(
      `/admin/llm-config/connections/${encodeURIComponent(connId)}${
        force ? "?force=true" : ""
      }`,
    );
  }

  /** Assign a connection (+ model) to a role — V2 shape {conn_id, model}. */
  updateLlmConfig(role: LlmRole, payload: RoleAssignPayload) {
    return this.put<RoleAssignResult>(`/admin/llm-config/${role}`, payload);
  }

  /** Clear a role's assignment override → back to .env defaults. */
  deleteLlmConfigOverride(role: LlmRole) {
    return this.delete<{ role: string; cleared: boolean }>(
      `/admin/llm-config/${role}`,
    );
  }

  /** Test a full candidate config before saving. Server pings the real endpoint. */
  testLlmConfig(payload: LlmTestRequest) {
    return this.post<LlmTestResult>("/admin/llm-config/test", payload);
  }

  /** Light model listing only — no completion ping, no save.
   *  Accepts {conn_id} or raw {provider, base_url, api_key}. */
  listLlmModels(payload: LlmModelsRequest) {
    return this.post<LlmModelsResponse>("/admin/llm-config/models", payload);
  }

  /** Model catalogue for one stored connection (credentials resolved server-side). */
  listLlmModelsByConn(connId: string) {
    return this.listLlmModels({ conn_id: connId });
  }

  async uploadFile<T>(
    path: string,
    file: File,
    extraHeaders?: Record<string, string>,
  ): Promise<T> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: {
        ...getAuthHeaders(),
        ...extraHeaders,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  /**
   * Upload a file directly to MinIO via a presigned URL (bypasses FastAPI for
   * the file bytes).
   *
   * Flow:
   *   1. POST /documents/upload/{wsId}/presign  → { document_id, upload_url, minio_key }
   *   2. PUT  upload_url (file bytes)           → 200 from MinIO
   *   3. POST /documents/upload/{wsId}/confirm  → DocumentUploadResponse
   */
  async uploadFileDirect<T>(
    workspaceId: string | number,
    file: File,
    onProgress?: (percent: number) => void,
  ): Promise<T> {
    // Step 1 — get presigned URL
    const presignRes = await fetch(`${BASE_URL}/documents/upload/${workspaceId}/presign`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeaders() },
      body: JSON.stringify({
        filename: file.name,
        file_size: file.size,
        content_type: file.type || undefined,
      }),
    });

    if (!presignRes.ok) {
      const err = await presignRes.json().catch(() => ({ detail: "Presign failed" }));
      throw new Error(err.detail || `Presign error: ${presignRes.status}`);
    }

    const { document_id, upload_url } = await presignRes.json() as {
      document_id: number;
      upload_url: string;
      minio_key: string;
    };

    // Rewrite presigned URL for Vite dev proxy (no-op in production)
    const putUrl = rewritePresignedUrl(upload_url);

    // Step 2 — PUT file bytes directly to MinIO (with optional XHR progress)
    await new Promise<void>((resolve, reject) => {
      if (onProgress) {
        const xhr = new XMLHttpRequest();
        xhr.open("PUT", putUrl);
        xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(`MinIO PUT failed: ${xhr.status}`)));
        xhr.onerror = () => reject(new Error("Network error during MinIO upload"));
        xhr.send(file);
      } else {
        fetch(putUrl, {
          method: "PUT",
          headers: { "Content-Type": file.type || "application/octet-stream" },
          body: file,
        })
          .then((r) => (r.ok ? resolve() : reject(new Error(`MinIO PUT failed: ${r.status}`))))
          .catch(reject);
      }
    });

    // Step 3 — confirm and trigger pipeline
    const confirmRes = await fetch(`${BASE_URL}/documents/upload/${workspaceId}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeaders() },
      body: JSON.stringify({ document_id }),
    });

    if (!confirmRes.ok) {
      const err = await confirmRes.json().catch(() => ({ detail: "Confirm failed" }));
      throw new Error(err.detail || `Confirm error: ${confirmRes.status}`);
    }

    return confirmRes.json();
  }
}

export const api = new ApiClient();
