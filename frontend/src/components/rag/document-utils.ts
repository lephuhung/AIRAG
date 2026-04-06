import { 
  FileText, 
  FileType, 
  Presentation, 
  FileCode, 
  Hash, 
  FileDigit,
  File,
  Clock,
  Loader2,
  CheckCircle2,
  XCircle
} from "lucide-react";
import type { DocumentStatus } from "@/types";

export const FILE_TYPE_CONFIG: Record<string, { icon: any; color: string; bgColor: string }> = {
  pdf:  { icon: FileText, color: "text-red-500", bgColor: "bg-red-500/10" },
  docx: { icon: FileType, color: "text-blue-500", bgColor: "bg-blue-500/10" },
  doc:  { icon: FileType, color: "text-blue-500", bgColor: "bg-blue-500/10" },
  pptx: { icon: Presentation, color: "text-orange-500", bgColor: "bg-orange-500/10" },
  ppt:  { icon: Presentation, color: "text-orange-500", bgColor: "bg-orange-500/10" },
  txt:  { icon: FileCode, color: "text-zinc-500", bgColor: "bg-zinc-500/10" },
  md:   { icon: Hash, color: "text-indigo-500", bgColor: "bg-indigo-500/10" },
  xlsx: { icon: FileDigit, color: "text-emerald-500", bgColor: "bg-emerald-500/10" },
  xls:  { icon: FileDigit, color: "text-emerald-500", bgColor: "bg-emerald-500/10" },
};

export function getFileConfig(fileType: string) {
  const ext = fileType.replace(".", "").toLowerCase();
  return FILE_TYPE_CONFIG[ext] ?? { icon: File, color: "text-muted-foreground", bgColor: "bg-muted/15" };
}

export const STATUS_CONFIG: Record<DocumentStatus, { labelKey: string; className: string; icon: any }> = {
  pending:      { labelKey: "files.status.pending",      className: "bg-muted text-muted-foreground",         icon: Clock },
  parsing:      { labelKey: "files.status.parsing",      className: "bg-blue-400/15 text-blue-400",           icon: Loader2 },
  ocring:       { labelKey: "files.status.ocring",       className: "bg-indigo-400/15 text-indigo-400",       icon: Loader2 },
  chunking:     { labelKey: "files.status.chunking",     className: "bg-cyan-400/15 text-cyan-400",           icon: Loader2 },
  embedding:    { labelKey: "files.status.embedding",    className: "bg-amber-400/15 text-amber-400",         icon: Loader2 },
  building_kg:  { labelKey: "files.status.building_kg",  className: "bg-violet-400/15 text-violet-400",       icon: Loader2 },
  indexed:      { labelKey: "files.status.indexed",      className: "bg-primary/15 text-primary",             icon: CheckCircle2 },
  failed:       { labelKey: "files.status.failed",       className: "bg-destructive/15 text-destructive",     icon: XCircle },
};
