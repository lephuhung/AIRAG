import { memo } from "react";
import { motion } from "framer-motion";
import { useTranslation } from "@/hooks/useTranslation";
import { Loader2, UploadCloud } from "lucide-react";
import type { UploadingFile } from "@/types";

interface UploadingCardProps {
  file: UploadingFile;
}

export const UploadingCard = memo(({ file }: UploadingCardProps) => {
  const { t } = useTranslation();
  
  const sizeStr = file.size >= 1024 * 1024
    ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.round(file.size / 1024)} KB`;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="relative rounded-lg border-2 border-primary/20 bg-primary/[0.03] px-4 py-3 shadow-md overflow-hidden"
    >

      <div className="flex items-center gap-3">
        {/* Animated Icon */}
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0 transition-colors">
          <UploadCloud className="w-5 h-5 text-primary animate-pulse" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <p className="font-semibold text-sm truncate">{file.name}</p>
            <span className="text-[10px] font-black text-primary uppercase tabular-nums bg-primary/10 px-1.5 py-0.5 rounded">
              {file.progress}%
            </span>
          </div>
          
          {/* Real progress bar */}
          <div className="w-full h-1.5 bg-primary/10 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${file.progress}%` }}
              className="h-full bg-primary shadow-[0_0_8px_rgba(59,130,246,0.5)]"
              transition={{ type: "spring", bounce: 0, duration: 0.5 }}
            />
          </div>

          <div className="flex items-center justify-between mt-0.5">
            <span className="text-[10px] text-muted-foreground font-medium">{sizeStr}</span>
            <span className="flex items-center gap-1.5 text-[10px] text-primary font-bold uppercase tracking-tight">
              <Loader2 className="w-3 h-3 animate-spin" />
              {t("workspace.uploading")}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
});
