// Small string helpers shared across the chat UI (extracted from ChatPanel.tsx).

export const truncateName = (name: string, maxLength = 25) => {
  if (name.length <= maxLength) return name;
  return name.slice(0, maxLength - 8) + "..." + name.slice(-5);
};

export const formatMentionName = (name: string) => {
  const clean = name.replace(/\.[^/.]+$/, "");
  return truncateName(clean, 30);
};

// Shorten filename for citation display (strips extension, adds ellipsis).
export function shortenDocName(filename: string, maxLen = 14): string {
  const name = filename.replace(/\.[^.]+$/, ""); // strip extension
  if (name.length <= maxLen) return name;
  return name.slice(0, maxLen - 1) + "…"; // ellipsis
}
