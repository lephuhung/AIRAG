const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

/**
 * Rewrite a presigned MinIO URL so it goes through the Vite dev proxy
 * instead of hitting localhost:9000 directly (which is blocked by CORS/network
 * in dev). In production (Docker/nginx) the URL is already reachable.
 *
 * localhost:9000/bucket/key?sig=... → /minio-direct/bucket/key?sig=...
 */
function rewritePresignedUrl(url: string): string {
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

class ApiClient {
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

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
    const response = await fetch(`${BASE_URL}${path}`, { method: "GET" });
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

  delete(path: string) {
    return this.request(path, { method: "DELETE" });
  }

  async downloadFile(path: string, filename: string): Promise<void> {
    const response = await fetch(`${BASE_URL}${path}`);

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

  async uploadFile<T>(path: string, file: File): Promise<T> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
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
    workspaceId: number,
    file: File,
    onProgress?: (percent: number) => void,
  ): Promise<T> {
    // Step 1 — get presigned URL
    const presignRes = await fetch(`${BASE_URL}/documents/upload/${workspaceId}/presign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
      headers: { "Content-Type": "application/json" },
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
