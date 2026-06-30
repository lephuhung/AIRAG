import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Image as ImageIcon } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";
import { useDocument } from "@/hooks/useDocuments";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import type { ChatImageRef } from "@/types";

// Image references panel — shows retrieved images in chat
function ImageRefCard({ img }: { img: ChatImageRef }) {
  const { activateImageCitation } = useWorkspaceStore();
  const { data: doc } = useDocument(img.document_id);
  return (
    <button
      onClick={() => activateImageCitation(img, doc)}
      className="group block rounded-md overflow-hidden border bg-background hover:border-primary/50 transition-colors text-left cursor-pointer"
    >
      <img
        src={img.url}
        alt={img.caption || `Image from page ${img.page_no}`}
        className="w-full h-auto max-h-[200px] object-contain bg-white"
        loading="lazy"
      />
      {img.caption && (
        <p className="px-2 py-1 text-[10px] text-muted-foreground leading-tight line-clamp-2 border-t">
          p.{img.page_no} — {img.caption}
        </p>
      )}
    </button>
  );
}

export function ImageRefsPanel({ images }: { images: ChatImageRef[] }) {
  const { t } = useTranslation();

  if (images.length === 0) return null;

  const [expanded, setExpanded] = useState(true);

  return (
    <div className="mt-2 rounded-md border bg-muted/20 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors"
      >
        <ImageIcon className="w-3 h-3" />
        {t("chat.images_from_docs", { count: images.length })}
        <span className="ml-auto text-[10px]">{expanded ? "▲" : "▼"}</span>
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: "auto" }}
            exit={{ height: 0 }}
            className="overflow-hidden border-t"
          >
            <div className="p-2 flex flex-wrap gap-2">
              {images.map((img, i) => (
                <ImageRefCard key={i} img={img} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
