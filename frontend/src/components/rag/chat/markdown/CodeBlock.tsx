import { useState, isValidElement, type ReactNode } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import css from "react-syntax-highlighter/dist/esm/languages/prism/css";
import markup from "react-syntax-highlighter/dist/esm/languages/prism/markup";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import cpp from "react-syntax-highlighter/dist/esm/languages/prism/cpp";
import diff from "react-syntax-highlighter/dist/esm/languages/prism/diff";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import { ClipboardCheck, Copy } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslation } from "@/hooks/useTranslation";
import { useThemeStore } from "@/stores/useThemeStore";

SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("ts", typescript);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("shell", bash);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("css", css);
SyntaxHighlighter.registerLanguage("html", markup);
SyntaxHighlighter.registerLanguage("xml", markup);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("yml", yaml);
SyntaxHighlighter.registerLanguage("java", java);
SyntaxHighlighter.registerLanguage("go", go);
SyntaxHighlighter.registerLanguage("cpp", cpp);
SyntaxHighlighter.registerLanguage("c", cpp);
SyntaxHighlighter.registerLanguage("diff", diff);
SyntaxHighlighter.registerLanguage("markdown", markdown);
SyntaxHighlighter.registerLanguage("md", markdown);

// Extract raw text from React node tree (for code blocks)
function extractText(node: ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (!node) return "";
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (isValidElement(node)) {
    const props = node.props as { children?: ReactNode };
    return extractText(props.children);
  }
  return "";
}

// Code block with syntax highlighting + copy button
export function CodeBlock({
  language,
  children,
}: {
  language: string;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const theme = useThemeStore((s) => s.theme);
  const isDark = theme === "dark";
  const code = extractText(children).replace(/\n$/, "");

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="group relative my-2">
      {language && (
        <span className="absolute top-2 right-2 text-[9px] uppercase text-muted-foreground/40 font-mono select-none z-10 pointer-events-none">
          {language}
        </span>
      )}
      <button
        onClick={handleCopy}
        aria-label={t("chat.copy_code")}
        className={cn(
          "absolute top-2 left-2 p-1 rounded-md text-muted-foreground/50 hover:text-muted-foreground transition-all opacity-0 group-hover:opacity-100 z-10",
          isDark ? "bg-white/5 hover:bg-white/10" : "bg-black/5 hover:bg-black/10"
        )}
      >
        {copied ? (
          <ClipboardCheck className="w-3 h-3 text-emerald-500" />
        ) : (
          <Copy className="w-3 h-3" />
        )}
      </button>
      <SyntaxHighlighter
        language={language}
        style={isDark ? oneDark : oneLight}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: "8px",
          fontSize: "12px",
          padding: "10px 12px",
          ...(isDark
            ? {
              background: "oklch(0.18 0.015 155)",
              border: "1px solid oklch(0.30 0.025 155)",
            }
            : {
              background: "oklch(0.96 0.008 105)",
              border: "1px solid oklch(0.88 0.018 105)",
            }),
        }}
        codeTagProps={{ style: { fontFamily: '"IBM Plex Mono", "Fira Code", monospace' } }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
