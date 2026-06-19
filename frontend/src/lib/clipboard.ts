/**
 * copyToClipboard
 * ===============
 * Copy text to the clipboard with a fallback for insecure (HTTP) contexts.
 *
 * `navigator.clipboard` is only exposed over HTTPS or on localhost. When the app
 * is served over plain HTTP (e.g. http://rag.hatinh.local) it is `undefined`, so
 * `navigator.clipboard.writeText(...)` throws. We fall back to a hidden
 * <textarea> + `document.execCommand("copy")`, which still works on HTTP.
 *
 * Returns `true` on success so callers can toast accordingly.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Secure API present but blocked (permissions/focus) — fall through to legacy.
  }

  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.top = "0";
    ta.style.left = "0";
    ta.style.opacity = "0";
    ta.style.pointerEvents = "none";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
