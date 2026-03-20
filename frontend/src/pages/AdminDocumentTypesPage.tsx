import { useState } from "react";
import { toast } from "sonner";
import {
  FileText,
  Plus,
  Search,
  Loader2,
  Pencil,
  ToggleLeft,
  ToggleRight,
  ChevronDown,
  ChevronUp,
  Save,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  useDocumentTypes,
  useCreateDocumentType,
  useUpdateDocumentType,
  useDeactivateDocumentType,
  useDocumentTypeGlobalPrompt,
  useSetGlobalPrompt,
} from "@/hooks/useDocumentTypes";
import type { DocumentTypeDetail } from "@/types";

export function AdminDocumentTypesPage() {
  const [search, setSearch] = useState("");
  const [includeInactive, setIncludeInactive] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [editTarget, setEditTarget] = useState<DocumentTypeDetail | null>(null);
  const [promptTarget, setPromptTarget] = useState<DocumentTypeDetail | null>(null);

  const { data: types, isLoading } = useDocumentTypes(includeInactive);
  const createType = useCreateDocumentType();
  const updateType = useUpdateDocumentType();
  const deactivate = useDeactivateDocumentType();

  const filtered = (types ?? []).filter(
    (t) =>
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      t.slug.toLowerCase().includes(search.toLowerCase()),
  );

  // ── Create form state ──
  const [newSlug, setNewSlug] = useState("");
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");

  const handleCreate = async () => {
    if (!newSlug.trim() || !newName.trim()) {
      toast.error("Slug and name are required");
      return;
    }
    try {
      await createType.mutateAsync({
        slug: newSlug.trim().toLowerCase().replace(/\s+/g, "_"),
        name: newName.trim(),
        description: newDesc.trim() || undefined,
      });
      toast.success("Document type created");
      setNewSlug("");
      setNewName("");
      setNewDesc("");
      setShowCreate(false);
    } catch (err: any) {
      toast.error(err.message || "Failed to create document type");
    }
  };

  // ── Edit form state ──
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");

  const openEdit = (t: DocumentTypeDetail) => {
    setEditTarget(t);
    setEditName(t.name);
    setEditDesc(t.description ?? "");
    setPromptTarget(null);
  };

  const handleSaveEdit = async () => {
    if (!editTarget) return;
    try {
      await updateType.mutateAsync({
        slug: editTarget.slug,
        data: { name: editName.trim(), description: editDesc.trim() || undefined },
      });
      toast.success("Updated");
      setEditTarget(null);
    } catch (err: any) {
      toast.error(err.message || "Failed to update");
    }
  };

  const handleToggleActive = async (t: DocumentTypeDetail) => {
    try {
      if (t.is_active) {
        await deactivate.mutateAsync(t.slug);
        toast.success(`"${t.name}" deactivated`);
      } else {
        await updateType.mutateAsync({ slug: t.slug, data: { is_active: true } });
        toast.success(`"${t.name}" activated`);
      }
    } catch (err: any) {
      toast.error(err.message || "Failed to update status");
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <FileText className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Document Types</h1>
              <p className="text-sm text-muted-foreground">
                Manage document classifications and their system prompts
              </p>
            </div>
          </div>
          <Button size="sm" onClick={() => setShowCreate((v) => !v)}>
            <Plus className="w-4 h-4 mr-1.5" />
            New Type
          </Button>
        </div>

        {/* Create form */}
        {showCreate && (
          <div className="mb-5 rounded-xl border bg-card p-4 space-y-3">
            <p className="text-sm font-semibold">Create Document Type</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  Slug <span className="text-destructive">*</span>
                </label>
                <input
                  value={newSlug}
                  onChange={(e) => setNewSlug(e.target.value)}
                  placeholder="e.g. thong_tu"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  Name <span className="text-destructive">*</span>
                </label>
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g. Thông tư"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">
                Description
              </label>
              <input
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="Optional description"
                className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
            </div>
            <div className="flex items-center justify-end gap-2">
              <Button variant="ghost" size="sm" onClick={() => setShowCreate(false)}>
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleCreate}
                disabled={createType.isPending || !newSlug || !newName}
              >
                {createType.isPending && (
                  <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                )}
                Create
              </Button>
            </div>
          </div>
        )}

        {/* Search + filter */}
        <div className="flex items-center gap-3 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by name or slug..."
              className="w-full pl-9 pr-4 py-2 text-sm rounded-lg border bg-card focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
          </div>
          <label className="flex items-center gap-2 text-xs text-muted-foreground select-none cursor-pointer">
            <input
              type="checkbox"
              checked={includeInactive}
              onChange={(e) => setIncludeInactive(e.target.checked)}
              className="rounded"
            />
            Show inactive
          </label>
        </div>

        {/* List */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <FileText className="w-10 h-10 mb-3 opacity-30" />
            <p className="text-sm">No document types found</p>
          </div>
        ) : (
          <div className="border rounded-xl overflow-hidden divide-y">
            {filtered.map((t) => (
              <DocumentTypeRow
                key={t.slug}
                type={t}
                isEditing={editTarget?.slug === t.slug}
                isPromptOpen={promptTarget?.slug === t.slug}
                editName={editName}
                editDesc={editDesc}
                onEditNameChange={setEditName}
                onEditDescChange={setEditDesc}
                onOpenEdit={() => openEdit(t)}
                onCancelEdit={() => setEditTarget(null)}
                onSaveEdit={handleSaveEdit}
                isSaving={updateType.isPending}
                onToggleActive={() => handleToggleActive(t)}
                onTogglePrompt={() =>
                  setPromptTarget((prev) =>
                    prev?.slug === t.slug ? null : t,
                  )
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Row component ─────────────────────────────────────────────────────────────

function DocumentTypeRow({
  type,
  isEditing,
  isPromptOpen,
  editName,
  editDesc,
  onEditNameChange,
  onEditDescChange,
  onOpenEdit,
  onCancelEdit,
  onSaveEdit,
  isSaving,
  onToggleActive,
  onTogglePrompt,
}: {
  type: DocumentTypeDetail;
  isEditing: boolean;
  isPromptOpen: boolean;
  editName: string;
  editDesc: string;
  onEditNameChange: (v: string) => void;
  onEditDescChange: (v: string) => void;
  onOpenEdit: () => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  isSaving: boolean;
  onToggleActive: () => void;
  onTogglePrompt: () => void;
}) {
  return (
    <div className={cn("bg-card", !type.is_active && "opacity-60")}>
      {/* Main row */}
      <div className="flex items-center gap-3 px-4 py-3">
        {/* Status dot */}
        <span
          className={cn(
            "w-2 h-2 rounded-full flex-shrink-0",
            type.is_active ? "bg-green-500" : "bg-muted-foreground/40",
          )}
        />

        {/* Info */}
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <div className="flex flex-col sm:flex-row gap-2">
              <input
                value={editName}
                onChange={(e) => onEditNameChange(e.target.value)}
                placeholder="Name"
                className="flex-1 px-2 py-1 text-sm rounded border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                autoFocus
              />
              <input
                value={editDesc}
                onChange={(e) => onEditDescChange(e.target.value)}
                placeholder="Description (optional)"
                className="flex-1 px-2 py-1 text-sm rounded border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
            </div>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <p className="text-sm font-medium">{type.name}</p>
                <code className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono">
                  {type.slug}
                </code>
              </div>
              {type.description && (
                <p className="text-xs text-muted-foreground mt-0.5 truncate">
                  {type.description}
                </p>
              )}
            </>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {isEditing ? (
            <>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 px-2 text-xs text-destructive hover:bg-destructive/10"
                onClick={onCancelEdit}
              >
                <X className="w-3.5 h-3.5" />
              </Button>
              <Button
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={onSaveEdit}
                disabled={isSaving || !editName.trim()}
              >
                {isSaving ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Save className="w-3.5 h-3.5" />
                )}
              </Button>
            </>
          ) : (
            <>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
                onClick={onOpenEdit}
                title="Edit"
              >
                <Pencil className="w-3.5 h-3.5" />
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className={cn(
                  "h-7 px-2 text-xs",
                  type.is_active
                    ? "text-muted-foreground hover:text-amber-600 hover:bg-amber-500/10"
                    : "text-green-600 hover:bg-green-500/10",
                )}
                onClick={onToggleActive}
                title={type.is_active ? "Deactivate" : "Activate"}
              >
                {type.is_active ? (
                  <ToggleRight className="w-4 h-4" />
                ) : (
                  <ToggleLeft className="w-4 h-4" />
                )}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className={cn(
                  "h-7 px-2 text-xs gap-1",
                  isPromptOpen
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-foreground",
                )}
                onClick={onTogglePrompt}
                title="System Prompt"
              >
                <span className="text-[11px]">Prompt</span>
                {isPromptOpen ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
              </Button>
            </>
          )}
        </div>
      </div>

      {/* System prompt panel */}
      {isPromptOpen && <SystemPromptPanel slug={type.slug} name={type.name} />}
    </div>
  );
}

// ── System Prompt Panel ───────────────────────────────────────────────────────

function SystemPromptPanel({ slug, name }: { slug: string; name: string }) {
  const { data, isLoading } = useDocumentTypeGlobalPrompt(slug);
  const setPrompt = useSetGlobalPrompt();
  const [value, setValue] = useState<string | null>(null);

  const currentPrompt = value ?? data?.system_prompt ?? "";
  const isDirty = value !== null && value !== data?.system_prompt;

  const handleSave = async () => {
    try {
      await setPrompt.mutateAsync({ slug, system_prompt: currentPrompt });
      toast.success(`System prompt saved for "${name}"`);
      setValue(null);
    } catch (err: any) {
      toast.error(err.message || "Failed to save prompt");
    }
  };

  return (
    <div className="px-4 pb-4 border-t bg-muted/10">
      <div className="pt-3 space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-xs font-medium text-muted-foreground">
            Global System Prompt
            {data?.is_default && (
              <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-muted text-muted-foreground">
                using default
              </span>
            )}
          </p>
          {isDirty && (
            <Button
              size="sm"
              className="h-6 px-2.5 text-xs"
              onClick={handleSave}
              disabled={setPrompt.isPending}
            >
              {setPrompt.isPending ? (
                <Loader2 className="w-3 h-3 animate-spin mr-1" />
              ) : null}
              Save
            </Button>
          )}
        </div>
        {isLoading ? (
          <div className="flex items-center gap-2 py-2">
            <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Loading...</span>
          </div>
        ) : (
          <textarea
            value={currentPrompt}
            onChange={(e) => setValue(e.target.value)}
            rows={6}
            className="w-full px-3 py-2 text-xs font-mono rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30 resize-y"
            placeholder="Enter system prompt for this document type..."
          />
        )}
      </div>
    </div>
  );
}
