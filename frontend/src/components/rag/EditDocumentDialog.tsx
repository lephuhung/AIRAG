import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useTranslation } from "@/hooks/useTranslation";
import { Loader2, X } from "lucide-react";
import type { Document } from "@/types";

interface EditDocumentDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  document: Document | null;
  onSave: (documentId: number, data: { document_number?: string; signer_name?: string }) => Promise<void>;
}

export const EditDocumentDialog: React.FC<EditDocumentDialogProps> = ({
  open,
  onOpenChange,
  document,
  onSave,
}) => {
  const { t } = useTranslation();
  const [docNumber, setDocNumber] = useState("");
  const [signer, setSigner] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (document) {
      setDocNumber(document.document_number || "");
      setSigner(document.signer_name || "");
    }
  }, [document, open]);

  const handleSave = async () => {
    if (!document) return;
    setIsSaving(true);
    try {
      await onSave(document.id, {
        document_number: docNumber,
        signer_name: signer,
      });
      onOpenChange(false);
    } catch (error) {
      console.error("Failed to save document metadata:", error);
    } finally {
      setIsSaving(false);
    }
  };

  if (!open) return null;

  return (
    <>
      <div
        className="fixed inset-0 z-[60] bg-black/50 backdrop-blur-sm"
        onClick={() => onOpenChange(false)}
      />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-[70] w-full max-w-[425px] bg-card border rounded-2xl shadow-2xl p-6 space-y-6 animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold">{t("files.edit_metadata")}</h3>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 rounded-full"
            onClick={() => onOpenChange(false)}
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="docNumber" className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {t("files.metadata.document_number")}
            </Label>
            <Input
              id="docNumber"
              value={docNumber}
              onChange={(e) => setDocNumber(e.target.value)}
              placeholder="e.g. 123/QD-UBND"
              className="h-10"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="signer" className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {t("files.metadata.signer")}
            </Label>
            <Input
              id="signer"
              value={signer}
              onChange={(e) => setSigner(e.target.value)}
              placeholder="e.g. Nguyễn Văn A"
              className="h-10"
            />
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 pt-2">
          <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={isSaving}>
            {t("common.cancel")}
          </Button>
          <Button onClick={handleSave} disabled={isSaving} className="px-6 shadow-lg shadow-primary/20">
            {isSaving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {t("common.save")}
          </Button>
        </div>
      </div>
    </>
  );
};
