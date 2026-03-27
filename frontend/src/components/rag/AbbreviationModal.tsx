import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useTranslation } from "@/hooks/useTranslation";
import { Loader2, X } from "lucide-react";
import type { Abbreviation } from "@/types";

interface AbbreviationModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  abbreviation: Abbreviation | null;
  initialShortForm?: string;
  onSave: (data: { short_form: string; full_form: string; description?: string }) => Promise<void>;
  isPending: boolean;
}

export const AbbreviationModal: React.FC<AbbreviationModalProps> = ({
  open,
  onOpenChange,
  abbreviation,
  initialShortForm = "",
  onSave,
  isPending,
}) => {
  const { t } = useTranslation();
  const [shortForm, setShortForm] = useState("");
  const [fullForm, setFullForm] = useState("");
  const [description, setDescription] = useState("");

  useEffect(() => {
    if (abbreviation) {
      setShortForm(abbreviation.short_form);
      setFullForm(abbreviation.full_form || "");
      setDescription(abbreviation.description || "");
    } else {
      setShortForm(initialShortForm);
      setFullForm("");
      setDescription("");
    }
  }, [abbreviation, initialShortForm, open]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!shortForm || !fullForm) return;
    
    await onSave({
      short_form: shortForm,
      full_form: fullForm,
      description,
    });
    onOpenChange(false);
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
            <h3 className="text-lg font-bold">
              {abbreviation ? t("admin.abbreviations.edit") : t("admin.abbreviations.create")}
            </h3>
            <p className="text-sm text-muted-foreground mt-1">
              {t("admin.abbreviations.subtitle")}
            </p>
          </div>
          <button
            onClick={() => onOpenChange(false)}
            className="p-2 hover:bg-muted rounded-full transition-colors"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>

        <form onSubmit={handleSave} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="shortForm" className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {t("admin.abbreviations.table.short_form")} <span className="text-destructive">*</span>
            </Label>
            <Input
              id="shortForm"
              value={shortForm}
              onChange={(e) => setShortForm(e.target.value)}
              placeholder="e.g. BMNN"
              className="h-10"
              required
              autoFocus
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="fullForm" className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {t("admin.abbreviations.table.full_form")} <span className="text-destructive">*</span>
            </Label>
            <Input
              id="fullForm"
              value={fullForm}
              onChange={(e) => setFullForm(e.target.value)}
              placeholder="e.g. Bộ máy nhà nước"
              className="h-10"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="description" className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {t("admin.abbreviations.table.description")}
            </Label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              className="w-full flex min-h-[80px] rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-2">
            <Button variant="ghost" type="button" onClick={() => onOpenChange(false)} disabled={isPending}>
              {t("common.cancel")}
            </Button>
            <Button type="submit" disabled={isPending || !shortForm || !fullForm} className="px-6 shadow-lg shadow-primary/20">
              {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t("common.save")}
            </Button>
          </div>
        </form>
      </div>
    </>
  );
};
