import { useContext } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import type { ChatSourceChunk, ChatImageRef } from "@/types";
import { AllSourcesCtx } from "@/components/rag/chat/contexts";
import { injectCitations } from "@/components/rag/chat/markdown/citations";
import { preprocessMarkdown } from "@/components/rag/chat/markdown/text";
import { CodeBlock } from "@/components/rag/chat/markdown/CodeBlock";

// Markdown renderer with inline citation links + LaTeX + code blocks
export function MarkdownWithCitations({
  content,
  sources,
  relatedEntities,
  imageRefs,
}: {
  content: string;
  sources: ChatSourceChunk[];
  relatedEntities: string[];
  imageRefs?: ChatImageRef[];
}) {
  const processed = preprocessMarkdown(content);

  // Fallback: accumulated sources from all messages in the conversation.
  // When the model references citation IDs from previous answers (e.g. when
  // it didn't call search_documents), we can still render them as links.
  const allSources = useContext(AllSourcesCtx);

  // Create a wrapper component that injects citations into rendered children
  const withCitations = (Tag: string) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ({ children, ...props }: any) => {
      const injected = injectCitations(children, sources, relatedEntities, imageRefs, allSources);
      return <Tag {...props}>{injected}</Tag>;
    };
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        p: withCitations("p"),
        li: withCitations("li"),
        td: withCitations("td"),
        th: withCitations("th"),
        h1: withCitations("h1"),
        h2: withCitations("h2"),
        h3: withCitations("h3"),
        h4: withCitations("h4"),
        h5: withCitations("h5"),
        h6: withCitations("h6"),
        strong: withCitations("strong"),
        em: withCitations("em"),
        a: ({ href, children, ...props }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
            {injectCitations(children, sources, relatedEntities, imageRefs, allSources)}
          </a>
        ),
        // Code block — delegate to CodeBlock for syntax highlighting
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        code: ({ className, children, ...props }: any) => {
          const langMatch = /language-(\w+)/.exec(className || "");
          // Inline code (no language class)
          if (!langMatch) {
            return <code className={className} {...props}>{children}</code>;
          }
          // Fenced code block → syntax highlighted
          return <CodeBlock language={langMatch[1]}>{children}</CodeBlock>;
        },
      }}
    >
      {processed}
    </ReactMarkdown>
  );
}
