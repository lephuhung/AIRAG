import { useState } from "react";
import { useTranslation } from "@/hooks/useTranslation";
import { toast } from "sonner";
import {
  BookMarked,
  Plus,
  Search,
  ChevronLeft,
  ChevronRight,
  Trash2,
  CheckCircle2,
  XCircle,
  Loader2,
  Edit2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/authStore";
import {
  useAbbreviations,
  useCreateAbbreviation,
  useUpdateAbbreviation,
  useDeleteAbbreviation,
} from "@/hooks/useAbbreviations";
import type { Abbreviation } from "@/types";

type FilterStatus = "all" | "active" | "pending";

export function AdminAbbreviationsPage() {
  const { t } = useTranslation();
  const currentUser = useAuthStore((s) => s.user);
  
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all");
  const [page, setPage] = useState(1);
  const perPage = 20;

  // Debounce search
  const [searchTimeout, setSearchTimeout] = useState<ReturnType<typeof setTimeout> | null>(null);
  const handleSearchChange = (value: string) => {
    setSearch(value);
    if (searchTimeout) clearTimeout(searchTimeout);
    const timeout = setTimeout(() => {
      setDebouncedSearch(value);
      setPage(1);
    }, 300);
    setSearchTimeout(timeout);
  };

  const isActiveFilter = filterStatus === "all" ? null : filterStatus === "active";

  const { data, isLoading } = useAbbreviations(
    debouncedSearch || undefined,
    isActiveFilter,
    page,
    perPage
  );

  const createAbb = useCreateAbbreviation();
  const updateAbb = useUpdateAbbreviation();
  const deleteAbb = useDeleteAbbreviation();

  const [confirmDelete, setConfirmDelete] = useState<Abbreviation | null>(null);
  const [editingItem, setEditingItem] = useState<Abbreviation | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // Form state
  const [formShort, setFormShort] = useState("");
  const [formFull, setFormFull] = useState("");
  const [formDesc, setFormDesc] = useState("");

  const totalPages = data ? Math.ceil(data.total / perPage) : 1;

  const handleOpenCreate = () => {
    setEditingItem(null);
    setFormShort("");
    setFormFull("");
    setFormDesc("");
    setIsModalOpen(true);
  };

  const handleOpenEdit = (item: Abbreviation) => {
    setEditingItem(item);
    setFormShort(item.short_form);
    setFormFull(item.full_form);
    setFormDesc(item.description || "");
    setIsModalOpen(true);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formShort || !formFull) {
      toast.error(t("admin.abbreviations.toast.required"));
      return;
    }

    try {
      if (editingItem) {
        await updateAbb.mutateAsync({
          id: editingItem.id,
          data: {
            short_form: formShort,
            full_form: formFull,
            description: formDesc,
          },
        });
        toast.success(t("admin.abbreviations.toast.updated"));
      } else {
        await createAbb.mutateAsync({
          short_form: formShort,
          full_form: formFull,
          description: formDesc,
        });
        toast.success(t("admin.abbreviations.toast.created"));
      }
      setIsModalOpen(false);
    } catch (err: any) {
      toast.error(err.message || t("admin.abbreviations.toast.error"));
    }
  };

  const handleToggleActive = async (item: Abbreviation) => {
    try {
      await updateAbb.mutateAsync({
        id: item.id,
        data: { is_active: !item.is_active },
      });
      const msg = item.is_active 
        ? t("admin.abbreviations.toast.deactivated", { short: item.short_form }) 
        : t("admin.abbreviations.toast.activated", { short: item.short_form });
      toast.success(msg);
    } catch (err: any) {
      toast.error(err.message || t("admin.abbreviations.toast.error"));
    }
  };

  const handleDelete = async () => {
    if (!confirmDelete) return;
    try {
      await deleteAbb.mutateAsync(confirmDelete.id);
      toast.success(t("admin.abbreviations.toast.deleted"));
      setConfirmDelete(null);
    } catch (err: any) {
      toast.error(err.message || t("admin.abbreviations.toast.error"));
    }
  };

  if (!currentUser?.is_superadmin) {
    return <div className="p-8 text-center">{t("common.no_permission")}</div>;
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <BookMarked className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold">{t("admin.abbreviations.title")}</h1>
              <p className="text-sm text-muted-foreground">
                {t("admin.abbreviations.subtitle")}
              </p>
            </div>
          </div>
          <Button onClick={handleOpenCreate} className="flex items-center gap-2">
            <Plus className="w-4 h-4" />
            {t("admin.abbreviations.create")}
          </Button>
        </div>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row items-center gap-3 mb-4">
          <div className="relative flex-1 w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder={t("admin.abbreviations.search_placeholder")}
              className="w-full pl-9 pr-4 py-2 text-sm rounded-lg border bg-card focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
          </div>
          <div className="flex gap-1 rounded-lg border bg-card p-0.5">
            {(["all", "active", "pending"] as FilterStatus[]).map((f) => (
              <button
                key={f}
                onClick={() => {
                  setFilterStatus(f);
                  setPage(1);
                }}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize",
                  filterStatus === f
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {t("admin.abbreviations.status." + f)}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : !data || data.items.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground border rounded-xl bg-card/50">
            <BookMarked className="w-10 h-10 mb-3 opacity-20" />
            <p className="text-sm">{t("admin.abbreviations.no_items")}</p>
          </div>
        ) : (
          <div className="border rounded-xl overflow-hidden bg-card">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/30">
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                      {t("admin.abbreviations.table.short_form")}
                    </th>
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                      {t("admin.abbreviations.table.full_form")}
                    </th>
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                      {t("admin.abbreviations.table.description")}
                    </th>
                    <th className="px-4 py-3 text-center font-medium text-muted-foreground">
                      {t("admin.abbreviations.table.status")}
                    </th>
                    <th className="px-4 py-3 text-center font-medium text-muted-foreground">
                      {t("admin.abbreviations.table.actions")}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((item: Abbreviation) => (
                    <tr key={item.id} className="border-b last:border-0 hover:bg-muted/20 transition-colors">
                      <td className="px-4 py-3 font-bold text-primary">{item.short_form}</td>
                      <td className="px-4 py-3">{item.full_form}</td>
                      <td className="px-4 py-3 text-muted-foreground max-w-[200px] truncate">
                        {item.description || "-"}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider",
                            item.is_active
                              ? "bg-green-500/15 text-green-600 border border-green-500/20"
                              : "bg-amber-500/15 text-amber-600 border border-amber-500/20"
                          )}
                        >
                          {item.is_active ? t("admin.abbreviations.status.active") : t("admin.abbreviations.status.pending")}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-center gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            className={cn(
                              "h-8 w-8 p-0 transition-colors",
                              item.is_active ? "text-amber-600 hover:bg-amber-500/10" : "text-green-600 hover:bg-green-500/10"
                            )}
                            onClick={() => handleToggleActive(item)}
                            title={item.is_active ? t("admin.abbreviations.actions.deactivate") : t("admin.abbreviations.actions.activate")}
                          >
                            {item.is_active ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-8 w-8 p-0 text-muted-foreground hover:bg-muted"
                            onClick={() => handleOpenEdit(item)}
                            title={t("common.edit")}
                          >
                            <Edit2 className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-8 w-8 p-0 text-destructive hover:bg-destructive/10"
                            onClick={() => setConfirmDelete(item)}
                            title={t("admin.abbreviations.actions.delete")}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Pagination */}
        {data && totalPages > 1 && (
          <div className="flex items-center justify-between mt-4">
            <p className="text-xs text-muted-foreground">
              {t("admin.users.pagination", {
                start: (page - 1) * perPage + 1,
                end: Math.min(page * perPage, data.total),
                total: data.total,
              })}
            </p>
            <div className="flex items-center gap-1">
              <Button
                size="sm"
                variant="ghost"
                className="h-7 w-7 p-0"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <span className="text-xs px-2 text-muted-foreground font-medium">
                {page} / {totalPages}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 w-7 p-0"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Create/Edit Modal */}
      {isModalOpen && (
        <>
          <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm" onClick={() => setIsModalOpen(false)} />
          <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md bg-card border rounded-2xl shadow-2xl p-6 space-y-4 animate-in fade-in zoom-in duration-200">
            <div>
              <h3 className="text-lg font-bold">
                {editingItem ? t("admin.abbreviations.edit") : t("admin.abbreviations.create")}
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                {t("admin.abbreviations.subtitle")}
              </p>
            </div>

            <form onSubmit={handleSave} className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  {t("admin.abbreviations.table.short_form")} <span className="text-destructive">*</span>
                </label>
                <input
                  autoFocus
                  type="text"
                  value={formShort}
                  onChange={(e) => setFormShort(e.target.value)}
                  placeholder="e.g. NV"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  {t("admin.abbreviations.table.full_form")} <span className="text-destructive">*</span>
                </label>
                <input
                  type="text"
                  value={formFull}
                  onChange={(e) => setFormFull(e.target.value)}
                  placeholder="e.g. Nhân viên"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground">
                  {t("admin.abbreviations.table.description")}
                </label>
                <textarea
                  value={formDesc}
                  onChange={(e) => setFormDesc(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30 resize-none"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <Button variant="ghost" type="button" onClick={() => setIsModalOpen(false)}>
                  {t("common.cancel")}
                </Button>
                <Button type="submit" disabled={createAbb.isPending || updateAbb.isPending}>
                  {(createAbb.isPending || updateAbb.isPending) && (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  )}
                  {t("common.save")}
                </Button>
              </div>
            </form>
          </div>
        </>
      )}

      {/* Delete Confirmation */}
      <ConfirmDialog
        open={!!confirmDelete}
        onCancel={() => setConfirmDelete(null)}
        onConfirm={handleDelete}
        title={t("admin.abbreviations.delete_confirm_title")}
        message={t("admin.abbreviations.delete_confirm_msg", { short: confirmDelete?.short_form })}
        confirmLabel={t("common.delete")}
        variant="danger"
      />
    </div>
  );
}
