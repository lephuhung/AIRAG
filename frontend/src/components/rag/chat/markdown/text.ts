// Pure markdown text transforms (no JSX) extracted from ChatPanel.tsx.

// Preprocess markdown: fix common LLM output issues (header spacing, table
// spacing, single-line display math, memory section markers).
export function preprocessMarkdown(text: string): string {
  const lines = text.split("\n");
  const result: string[] = [];
  let prevWasTable = false;
  let inCodeFence = false;

  for (const line of lines) {
    let processedLine = line;
    // Fix: Headers lacking a space (e.g. "##Title" -> "## Title")
    if (/^#{1,6}[^#\s]/.test(processedLine)) {
      processedLine = processedLine.replace(/^(#{1,6})([^#\s])/, "$1 $2");
    }

    if (processedLine.trim().startsWith("```")) {
      inCodeFence = !inCodeFence;
    }

    const isTable = (processedLine.trim().startsWith("|") && processedLine.trim().endsWith("|")) ||
      /^\|[\s:|-]+\|$/.test(processedLine.trim());

    // Insert blank line before a table if needed
    if (isTable && !prevWasTable && result.length > 0 && result[result.length - 1].trim() !== "") {
      result.push("");
    }

    // Insert blank line after a table if current line is not a table
    if (prevWasTable && !isTable && processedLine.trim() !== "") {
      result.push("");
    }

    // Convert single-line display math $$content$$ to multi-line format
    if (
      !inCodeFence &&
      processedLine.trim().startsWith("$$") &&
      processedLine.trim().endsWith("$$") &&
      processedLine.trim().length > 4 &&
      processedLine.trim() !== "$$"
    ) {
      const mathContent = processedLine.trim().slice(2, -2);
      result.push("$$");
      result.push(mathContent);
      result.push("$$");
    } else {
      result.push(processedLine);
    }

    prevWasTable = isTable;
  }

  // Convert memory section markers to a styled markdown heading for ReactMarkdown.
  // Backend emits "[Memory]" (current) — also handle legacy "<memory_section>" tag.
  let processed = result.join("\n");
  processed = processed.replace(/\[Memory\]/gi, "\n---\n🧠 ");
  processed = processed.replace(/<memory_section>/gi, "\n---\n🧠 ");

  return processed;
}

// \s* sau "[": model đôi khi phát "[ a3f7]" có space — vẫn phải strip được
const CITATION_STRIP_RE = /\s*\[\s*(?:[a-z0-9]+|IMG-[a-z0-9]+)(?:,\s*(?:[a-z0-9]+|IMG-[a-z0-9]+))*\s*\]/g;

/** Remove citation references like [a3x9], [IMG-p4f2], [a3x9, b2m7] */
export function stripCitations(md: string): string {
  return md.replace(CITATION_STRIP_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/** Convert markdown to plain text: strip formatting, links, images, code fences */
export function markdownToPlainText(md: string): string {
  let text = stripCitations(md);
  text = text.replace(/```[\s\S]*?```/g, (m) => {
    const lines = m.split("\n");
    return lines.slice(1, -1).join("\n");
  });
  text = text.replace(/`([^`]+)`/g, "$1");
  text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1");
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  text = text.replace(/\*\*(.+?)\*\*/g, "$1");
  text = text.replace(/\*(.+?)\*/g, "$1");
  text = text.replace(/__(.+?)__/g, "$1");
  text = text.replace(/_(.+?)_/g, "$1");
  text = text.replace(/^#{1,6}\s+/gm, "");
  text = text.replace(/^[-*_]{3,}\s*$/gm, "");
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}
